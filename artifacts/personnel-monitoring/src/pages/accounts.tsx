import React, { useState } from "react";
import { AppLayout } from "@/components/layout/AppLayout";
import { useListAccounts, useDeleteAccount, useUpdateAccount } from "@workspace/api-client-react";
import { format } from "date-fns";
import { Shield, KeyRound, Trash2, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { useQueryClient } from "@tanstack/react-query";

export default function Accounts() {
  const queryClient = useQueryClient();
  const { data: accounts, isLoading } = useListAccounts();
  const deleteAccount = useDeleteAccount();
  const updateAccount = useUpdateAccount();
  const [searchTerm, setSearchTerm] = useState("");

  const [deleteId, setDeleteId] = useState<number | null>(null);
  
  const [updateId, setUpdateId] = useState<number | null>(null);
  const [newPassword, setNewPassword] = useState("");

  const filteredAccounts = accounts?.filter(acc => 
    acc.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (acc.personnelName && acc.personnelName.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const handleDelete = () => {
    if (deleteId) {
      deleteAccount.mutate({ id: deleteId }, {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["/api/accounts"] });
          setDeleteId(null);
        }
      });
    }
  };

  const handleUpdatePassword = () => {
    if (updateId && newPassword.length >= 6) {
      updateAccount.mutate({ id: updateId, data: { password: newPassword } }, {
        onSuccess: () => {
          setUpdateId(null);
          setNewPassword("");
          // Could add a toast notification here
        }
      });
    }
  };

  return (
    <AppLayout>
      <div className="max-w-6xl mx-auto flex flex-col h-[calc(100vh-8rem)]">
        <div className="mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 tracking-tight">Manage Accounts</h2>
            <p className="text-gray-500 mt-1">Control system access and reset passwords.</p>
          </div>
          <div className="relative w-full sm:w-80">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <Input 
              placeholder="Search accounts..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9 bg-white shadow-sm"
            />
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-2xl shadow-sm flex-1 overflow-hidden flex flex-col">
          <div className="flex-1 overflow-auto">
            {isLoading ? (
              <div className="h-full flex items-center justify-center">
                <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
              </div>
            ) : (
              <table className="w-full text-left border-collapse">
                <thead className="bg-[#f8f9fa] sticky top-0 shadow-sm z-10">
                  <tr>
                    <th className="py-4 px-6 font-bold text-gray-700 text-xs uppercase tracking-wider">User</th>
                    <th className="py-4 px-6 font-bold text-gray-700 text-xs uppercase tracking-wider">Role</th>
                    <th className="py-4 px-6 font-bold text-gray-700 text-xs uppercase tracking-wider">Linked Personnel</th>
                    <th className="py-4 px-6 font-bold text-gray-700 text-xs uppercase tracking-wider">Created</th>
                    <th className="py-4 px-6 font-bold text-gray-700 text-xs uppercase tracking-wider text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {filteredAccounts?.map((account) => (
                    <tr key={account.id} className="hover:bg-gray-50 transition-colors">
                      <td className="py-4 px-6">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded bg-gray-100 border border-gray-200 flex items-center justify-center">
                            <Shield className={`w-4 h-4 ${account.role === 'admin' ? 'text-primary' : 'text-blue-500'}`} />
                          </div>
                          <span className="font-bold text-gray-900">{account.username}</span>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider border ${
                          account.role === 'admin' ? 'bg-red-50 text-red-700 border-red-100' : 'bg-blue-50 text-blue-700 border-blue-100'
                        }`}>
                          {account.role}
                        </span>
                      </td>
                      <td className="py-4 px-6 text-sm text-gray-600 font-medium">
                        {account.personnelName || <span className="text-gray-400 italic">None</span>}
                      </td>
                      <td className="py-4 px-6 text-sm text-gray-500">
                        {format(new Date(account.createdAt), 'MMM d, yyyy')}
                      </td>
                      <td className="py-4 px-6 text-right">
                        <div className="flex justify-end gap-2">
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="h-8"
                            onClick={() => setUpdateId(account.id)}
                          >
                            <KeyRound className="w-4 h-4 mr-1" />
                            Reset Pass
                          </Button>
                          {account.role !== 'admin' && ( // Prevent deleting other admins easily for safety in this UI
                            <Button 
                              variant="destructive" 
                              size="icon" 
                              className="h-8 w-8"
                              onClick={() => setDeleteId(account.id)}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog 
        open={deleteId !== null} 
        onOpenChange={(open) => !open && setDeleteId(null)}
        title="Delete Account"
        description="Are you sure you want to delete this account? This action cannot be undone."
      >
        <div className="mt-6 flex justify-end gap-3">
          <Button variant="outline" onClick={() => setDeleteId(null)}>Cancel</Button>
          <Button variant="destructive" onClick={handleDelete} disabled={deleteAccount.isPending}>
            {deleteAccount.isPending ? "Deleting..." : "Yes, Delete"}
          </Button>
        </div>
      </Dialog>

      {/* Update Password Dialog */}
      <Dialog 
        open={updateId !== null} 
        onOpenChange={(open) => {
          if (!open) { setUpdateId(null); setNewPassword(""); }
        }}
        title="Reset Password"
        description="Enter a new password for this user account. Must be at least 6 characters."
      >
        <div className="mt-4 space-y-4">
          <div>
            <Input 
              type="password" 
              placeholder="New Password" 
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="outline" onClick={() => { setUpdateId(null); setNewPassword(""); }}>Cancel</Button>
            <Button 
              onClick={handleUpdatePassword} 
              disabled={newPassword.length < 6 || updateAccount.isPending}
            >
              {updateAccount.isPending ? "Saving..." : "Save Password"}
            </Button>
          </div>
        </div>
      </Dialog>
    </AppLayout>
  );
}
